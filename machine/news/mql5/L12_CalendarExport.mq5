//+------------------------------------------------------------------+
//| L12_CalendarExport.mq5 - ML Bot, prompt L12 (the news calendar)  |
//|                                                                  |
//| Exports MT5's economic calendar for one country to text files.   |
//| READ ONLY: it sends no order and reads no price.                 |
//|                                                                  |
//| Two files, split on InpSealedFrom (TRADE SERVER time):           |
//|   calendar_<country>_open.tsv    rows before it (2015-2023)      |
//|   calendar_<country>_sealed.tsv  rows from it on (2024+). The    |
//|     machine copies this one unread: 2024-2026 are sealed         |
//|     (LEARNED #96 "No, don't touch those years").                 |
//| Then calendar_<country>_info.txt, written LAST (= done).         |
//|                                                                  |
//| Every event of every importance is written; the machine keeps    |
//| high + moderate (Forex Factory red + orange, T7).                |
//| Numbers are written as MT5 stores them: the value x 1,000,000.   |
//| An empty cell = MT5 has no value (LONG_MIN).                     |
//| Times: TRADE SERVER time, as MT5's calendar functions give them. |
//|                                                                  |
//| Run by: python -m machine.news.mt5_export (it copies this file   |
//| into the terminal, compiles it and starts it).                   |
//+------------------------------------------------------------------+
#property copyright "ML Bot"
#property version   "1.00"

input string   InpCountry    = "US";
input datetime InpFrom       = D'2015.01.01 00:00';
input datetime InpSealedFrom = D'2024.01.01 00:00';
input string   InpFolder     = "ML-Bot";

string Cell(const long v)
  {
   return v == LONG_MIN ? "" : IntegerToString(v);
  }

string Clean(string s)
  {
   StringReplace(s, "\t", " ");
   StringReplace(s, "\r", " ");
   StringReplace(s, "\n", " ");
   return s;
  }

string Stamp(const datetime t)
  {
   return TimeToString(t, TIME_DATE | TIME_SECONDS);
  }

string Header()
  {
   return "value_id\tevent_id\tevent_code\tevent_name\tevent_type\tsector\tfrequency\ttime_mode\timportance"
          "\tunit\tmultiplier\tdigits\ttime_server\tperiod\trevision\tactual_x1e6\tforecast_x1e6\tprev_x1e6"
          "\trevised_prev_x1e6\timpact_type\tsource_url";
  }

int WriteRows(const int handle, const MqlCalendarEvent &ev, const MqlCalendarValue &vals[])
  {
   int n = ArraySize(vals);
   for(int k = 0; k < n; k++)
     {
      string line = StringFormat("%I64u\t%I64u\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%s",
                                 vals[k].id, ev.id, Clean(ev.event_code), Clean(ev.name), (int)ev.type,
                                 (int)ev.sector, (int)ev.frequency, (int)ev.time_mode, (int)ev.importance,
                                 (int)ev.unit, (int)ev.multiplier, ev.digits, Stamp(vals[k].time),
                                 Stamp(vals[k].period), vals[k].revision, Cell(vals[k].actual_value),
                                 Cell(vals[k].forecast_value), Cell(vals[k].prev_value),
                                 Cell(vals[k].revised_prev_value), (int)vals[k].impact_type, Clean(ev.source_url));
      FileWriteString(handle, line + "\n");
     }
   return n;
  }

int OpenOut(const string name)
  {
   int h = FileOpen(name, FILE_WRITE | FILE_TXT | FILE_ANSI, '\t', CP_UTF8);
   if(h == INVALID_HANDLE)
      PrintFormat("L12: cannot open %s (error %d)", name, GetLastError());
   return h;
  }

void OnStart()
  {
   string base = InpFolder + "\\calendar_" + InpCountry;
   string info = base + "_info.txt";
   FileDelete(info);

   // The terminal may still be loading the calendar from its server right after it starts: wait up to 2 minutes.
   MqlCalendarEvent events[];
   int n = 0;
   for(int attempt = 0; attempt < 24 && n <= 0 && !IsStopped(); attempt++)
     {
      ResetLastError();
      n = CalendarEventByCountry(InpCountry, events);
      if(n <= 0)
        {
         PrintFormat("L12: no calendar events for %s yet (error %d) - waiting 5 s", InpCountry, GetLastError());
         Sleep(5000);
        }
     }

   int hOpen = OpenOut(base + "_open.tmp");
   int hSealed = OpenOut(base + "_sealed.tmp");
   if(hOpen == INVALID_HANDLE || hSealed == INVALID_HANDLE)
      return;
   FileWriteString(hOpen, Header() + "\n");
   FileWriteString(hSealed, Header() + "\n");

   int rowsOpen = 0, rowsSealed = 0, failed = 0;
   int byImportance[4] = {0, 0, 0, 0};
   for(int i = 0; i < n && !IsStopped(); i++)
     {
      MqlCalendarValue vals[];
      ResetLastError();
      // "to" is inclusive: stop one second before the sealed years.
      if(!CalendarValueHistoryByEvent(events[i].id, vals, InpFrom, InpSealedFrom - 1) && GetLastError() != 0)
         failed++;
      int got = WriteRows(hOpen, events[i], vals);
      rowsOpen += got;
      int imp = (int)events[i].importance;
      if(imp >= 0 && imp < 4)
         byImportance[imp] += got;

      MqlCalendarValue later[];
      ResetLastError();
      if(!CalendarValueHistoryByEvent(events[i].id, later, InpSealedFrom, 0) && GetLastError() != 0)
         failed++;
      rowsSealed += WriteRows(hSealed, events[i], later);
     }
   FileClose(hOpen);
   FileClose(hSealed);
   FileMove(base + "_open.tmp", 0, base + "_open.tsv", FILE_REWRITE);
   FileMove(base + "_sealed.tmp", 0, base + "_sealed.tsv", FILE_REWRITE);

   int h = OpenOut(info);
   if(h == INVALID_HANDLE)
      return;
   datetime server = TimeTradeServer();
   datetime gmt = TimeGMT();
   FileWriteString(h, "server=" + AccountInfoString(ACCOUNT_SERVER) + "\n");
   FileWriteString(h, "company=" + AccountInfoString(ACCOUNT_COMPANY) + "\n");
   FileWriteString(h, "terminal_build=" + IntegerToString(TerminalInfoInteger(TERMINAL_BUILD)) + "\n");
   FileWriteString(h, "exported_at_server=" + Stamp(server) + "\n");
   FileWriteString(h, "exported_at_gmt=" + Stamp(gmt) + "\n");
   FileWriteString(h, "server_minus_gmt_seconds=" + IntegerToString((long)(server - gmt)) + "\n");
   FileWriteString(h, "country=" + InpCountry + "\n");
   FileWriteString(h, "from_server=" + Stamp(InpFrom) + "\n");
   FileWriteString(h, "sealed_from_server=" + Stamp(InpSealedFrom) + "\n");
   FileWriteString(h, "events=" + IntegerToString(n) + "\n");
   FileWriteString(h, "rows_open=" + IntegerToString(rowsOpen) + "\n");
   FileWriteString(h, StringFormat("rows_open_by_importance=none:%d low:%d moderate:%d high:%d\n",
                                   byImportance[0], byImportance[1], byImportance[2], byImportance[3]));
   FileWriteString(h, "rows_sealed=" + IntegerToString(rowsSealed) + "\n");
   FileWriteString(h, "failed_calls=" + IntegerToString(failed) + "\n");
   FileWriteString(h, "done=1\n");
   FileClose(h);
   PrintFormat("L12: %d events, %d rows before the sealed years, %d rows sealed, %d failed calls",
               n, rowsOpen, rowsSealed, failed);
  }
//+------------------------------------------------------------------+
